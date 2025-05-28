import pandas as pd
import numpy as np
from datetime import datetime
import os
from sqlalchemy import create_engine, text
import psycopg2
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_warehouse_log.log'),
        logging.StreamHandler()
    ]
)

class SalesDataWarehouse:
    def __init__(self):
        # Database connection string
        # # local
        self.connection_string = "postgresql://postgres:admin@localhost:5432/hmsales_dw"
        # # cloud
        # self.connection_string = "postgresql://postgres:uVekppVJDfDolutoNDclfuVysuIhiFfl@shinkansen.proxy.rlwy.net:29896/railway"
        self.engine = create_engine(self.connection_string)
        
    def test_connection(self):
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()
                logging.info(f"Connected to PostgreSQL: {version[0]}")
                return True
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            return False
    
    def create_database_schema(self):
        """Create database tables"""
        try:
            # SQL statements untuk membuat tabel
            create_tables_sql = """
            -- Drop tables if exist (untuk testing)
            DROP TABLE IF EXISTS fact_sales CASCADE;
            DROP TABLE IF EXISTS dimension_date CASCADE;
            DROP TABLE IF EXISTS dimension_product CASCADE;
            DROP TABLE IF EXISTS dimension_location CASCADE;
            DROP TABLE IF EXISTS dimension_order CASCADE;
            
            -- Create dimension_date table
            CREATE TABLE dimension_date (
                date_id SERIAL PRIMARY KEY,
                order_date DATE UNIQUE,
                day VARCHAR(2),
                month VARCHAR(2),
                month_name VARCHAR(20),
                quarter VARCHAR(2),
                year VARCHAR(4)
            );
            
            -- Create dimension_product table
            CREATE TABLE dimension_product (
                product_id VARCHAR(20) PRIMARY KEY,
                category VARCHAR(50),
                sub_category VARCHAR(50)
            );
            
            -- Create dimension_location table
            CREATE TABLE dimension_location (
                location_id SERIAL PRIMARY KEY,
                country VARCHAR(50),
                city VARCHAR(100),
                state VARCHAR(50),
                region VARCHAR(20),
                UNIQUE(country, city, state, region)
            );
            
            -- Create dimension_order table
            CREATE TABLE dimension_order (
                order_id VARCHAR(20) PRIMARY KEY,
                ship_mode VARCHAR(20),
                order_date DATE
            );
            
            -- Create fact_sales table
            CREATE TABLE fact_sales (
                row_id VARCHAR(20) PRIMARY KEY,
                customer_id VARCHAR(20),
                order_id VARCHAR(20),
                product_id VARCHAR(20),
                date_id INTEGER,
                location_id INTEGER,
                sales FLOAT,
                quantity INTEGER,
                discount FLOAT,
                profit FLOAT,
                FOREIGN KEY (order_id) REFERENCES dimension_order(order_id),
                FOREIGN KEY (product_id) REFERENCES dimension_product(product_id),
                FOREIGN KEY (date_id) REFERENCES dimension_date(date_id),
                FOREIGN KEY (location_id) REFERENCES dimension_location(location_id)
            );
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            
            logging.info("Database schema created successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to create database schema: {e}")
            return False
    
    def load_and_preprocess_data(self, csv_data):
        """Load and preprocess data from CSV content"""
        try:
            # Convert CSV string to DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(csv_data))
            
            logging.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Data cleaning and preprocessing
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            
            # Remove rows with missing critical data
            df = df.dropna(subset=['Order ID', 'Product ID', 'Customer ID'])
            
            # Data validation
            df = df[df['Sales'] >= 0]
            df = df[(df['Discount'] >= 0) & (df['Discount'] <= 1)]
            
            logging.info(f"Data after preprocessing: {len(df)} rows")
            return df
        except Exception as e:
            logging.error(f"Data preprocessing failed: {e}")
            return None
    
    def create_dimension_tables(self, df):
        """Create and populate dimension tables"""
        try:
            # 1. Dimension Date
            date_data = df[['Order Date']].drop_duplicates().copy()
            date_data['day'] = date_data['Order Date'].dt.day.astype(str)
            date_data['month'] = date_data['Order Date'].dt.month.astype(str)
            date_data['month_name'] = date_data['Order Date'].dt.strftime('%B')
            date_data['quarter'] = date_data['Order Date'].dt.quarter.astype(str)
            date_data['year'] = date_data['Order Date'].dt.year.astype(str)
            date_data.columns = ['order_date', 'day', 'month', 'month_name', 'quarter', 'year']
            
            # Insert dimension_date
            date_data.to_sql('dimension_date', self.engine, if_exists='append', index=False, method='multi')
            logging.info(f"Inserted {len(date_data)} records into dimension_date")
            
            # 2. Dimension Product
            product_data = df[['Product ID', 'Category', 'Sub-Category']].drop_duplicates().copy()
            product_data.columns = ['product_id', 'category', 'sub_category']
            
            product_data.to_sql('dimension_product', self.engine, if_exists='append', index=False, method='multi')
            logging.info(f"Inserted {len(product_data)} records into dimension_product")
            
            # 3. Dimension Location
            location_data = df[['Country', 'City', 'State', 'Region']].drop_duplicates().copy()
            location_data.columns = ['country', 'city', 'state', 'region']
            
            location_data.to_sql('dimension_location', self.engine, if_exists='append', index=False, method='multi')
            logging.info(f"Inserted {len(location_data)} records into dimension_location")
            
            # 4. Dimension Order
            order_data = df[['Order ID', 'Ship Mode', 'Order Date']].drop_duplicates().copy()
            order_data.columns = ['order_id', 'ship_mode', 'order_date']
            
            order_data.to_sql('dimension_order', self.engine, if_exists='append', index=False, method='multi')
            logging.info(f"Inserted {len(order_data)} records into dimension_order")
            
            return True
        except Exception as e:
            logging.error(f"Failed to create dimension tables: {e}")
            return False
    
    def create_fact_table(self, df):
        """Create and populate fact table"""
        try:
            # Prepare fact data
            fact_data = df.copy()
            fact_data['row_id'] = ['ROW_' + str(i+1).zfill(6) for i in range(len(fact_data))]
            
            # Get foreign keys
            # Date IDs
            with self.engine.connect() as conn:
                date_mapping = pd.read_sql(
                    "SELECT date_id, order_date FROM dimension_date", 
                    conn
                )
                date_mapping['order_date'] = pd.to_datetime(date_mapping['order_date'])
                date_dict = dict(zip(date_mapping['order_date'], date_mapping['date_id']))
                
                # Location IDs
                location_mapping = pd.read_sql(
                    "SELECT location_id, country, city, state, region FROM dimension_location", 
                    conn
                )
            
            # Map date_id
            fact_data['date_id'] = fact_data['Order Date'].map(date_dict)
            
            # Map location_id
            location_dict = {}
            for _, row in location_mapping.iterrows():
                key = (row['country'], row['city'], row['state'], row['region'])
                location_dict[key] = row['location_id']
            
            fact_data['location_id'] = fact_data.apply(
                lambda row: location_dict.get((row['Country'], row['City'], row['State'], row['Region'])), 
                axis=1
            )
            
            # Select final columns for fact table
            fact_final = fact_data[[
                'row_id', 'Customer ID', 'Order ID', 'Product ID', 
                'date_id', 'location_id', 'Sales', 'Quantity', 'Discount', 'Profit'
            ]].copy()
            
            fact_final.columns = [
                'row_id', 'customer_id', 'order_id', 'product_id', 
                'date_id', 'location_id', 'sales', 'quantity', 'discount', 'profit'
            ]
            
            # Remove rows with missing foreign keys
            fact_final = fact_final.dropna(subset=['date_id', 'location_id'])
            
            # Insert fact data
            fact_final.to_sql('fact_sales', self.engine, if_exists='append', index=False, method='multi')
            logging.info(f"Inserted {len(fact_final)} records into fact_sales")
            
            return True
        except Exception as e:
            logging.error(f"Failed to create fact table: {e}")
            return False
    
    def create_indexes(self):
        """Create database indexes for performance"""
        try:
            index_sql = """
            -- Indexes for dimension tables
            CREATE INDEX IF NOT EXISTS idx_dimension_date_year ON dimension_date(year);
            CREATE INDEX IF NOT EXISTS idx_dimension_date_month ON dimension_date(month);
            CREATE INDEX IF NOT EXISTS idx_dimension_date_quarter ON dimension_date(quarter);
            
            CREATE INDEX IF NOT EXISTS idx_dimension_product_category ON dimension_product(category);
            CREATE INDEX IF NOT EXISTS idx_dimension_product_subcategory ON dimension_product(sub_category);
            
            CREATE INDEX IF NOT EXISTS idx_dimension_location_region ON dimension_location(region);
            CREATE INDEX IF NOT EXISTS idx_dimension_location_state ON dimension_location(state);
            
            CREATE INDEX IF NOT EXISTS idx_dimension_order_shipmode ON dimension_order(ship_mode);
            
            -- Indexes for fact table
            CREATE INDEX IF NOT EXISTS idx_fact_sales_order_id ON fact_sales(order_id);
            CREATE INDEX IF NOT EXISTS idx_fact_sales_product_id ON fact_sales(product_id);
            CREATE INDEX IF NOT EXISTS idx_fact_sales_date_id ON fact_sales(date_id);
            CREATE INDEX IF NOT EXISTS idx_fact_sales_location_id ON fact_sales(location_id);
            CREATE INDEX IF NOT EXISTS idx_fact_sales_customer_id ON fact_sales(customer_id);
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(index_sql))
                conn.commit()
            
            logging.info("Database indexes created successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to create indexes: {e}")
            return False
    
    def create_analysis_views(self):
        """Create views for easy analysis"""
        try:
            views_sql = """
            -- Sales summary view
            CREATE OR REPLACE VIEW v_sales_summary AS
            SELECT 
                fs.row_id,
                fs.customer_id,
                dord.order_id,
                dord.ship_mode,
                dp.product_id,
                dp.category,
                dp.sub_category,
                dd.order_date,
                dd.year,
                dd.month_name,
                dd.quarter,
                dl.country,
                dl.city,
                dl.state,
                dl.region,
                fs.sales,
                fs.quantity,
                fs.discount,
                fs.profit,
                CASE 
                    WHEN fs.sales > 0 THEN (fs.profit / fs.sales) * 100 
                    ELSE 0 
                END as profit_margin_pct
            FROM fact_sales fs
            JOIN dimension_order dord ON fs.order_id = dord.order_id
            JOIN dimension_product dp ON fs.product_id = dp.product_id
            JOIN dimension_date dd ON fs.date_id = dd.date_id
            JOIN dimension_location dl ON fs.location_id = dl.location_id;
            
            -- Monthly aggregation view
            CREATE OR REPLACE VIEW v_monthly_summary AS
            SELECT 
                dd.year,
                dd.month,
                dd.month_name,
                dp.category,
                dl.region,
                COUNT(fs.row_id) as total_orders,
                SUM(fs.sales) as total_sales,
                SUM(fs.quantity) as total_quantity,
                SUM(fs.profit) as total_profit,
                AVG(fs.profit) as avg_profit,
                AVG(CASE WHEN fs.sales > 0 THEN (fs.profit / fs.sales) * 100 ELSE 0 END) as avg_profit_margin
            FROM fact_sales fs
            JOIN dimension_date dd ON fs.date_id = dd.date_id
            JOIN dimension_product dp ON fs.product_id = dp.product_id
            JOIN dimension_location dl ON fs.location_id = dl.location_id
            GROUP BY dd.year, dd.month, dd.month_name, dp.category, dl.region;
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(views_sql))
                conn.commit()
            
            logging.info("Analysis views created successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to create views: {e}")
            return False
    
    def run_sample_analysis(self):
        """Run sample analysis queries"""
        try:
            logging.info("Running sample analysis queries...")
            
            queries = {
                "Total Sales by Category": """
                    SELECT category, SUM(sales) as total_sales 
                    FROM v_sales_summary 
                    GROUP BY category 
                    ORDER BY total_sales DESC
                """,
                "Monthly Sales Trend": """
                    SELECT year, month_name, SUM(total_sales) as monthly_sales 
                    FROM v_monthly_summary 
                    GROUP BY year, month_name, month 
                    ORDER BY year, month
                """,
                "Top 5 Cities by Sales": """
                    SELECT city, state, SUM(sales) as total_sales 
                    FROM v_sales_summary 
                    GROUP BY city, state 
                    ORDER BY total_sales DESC 
                    LIMIT 5
                """,
                "Profit Margin by Category": """
                    SELECT category, AVG(profit_margin_pct) as avg_profit_margin 
                    FROM v_sales_summary 
                    GROUP BY category 
                    ORDER BY avg_profit_margin DESC
                """
            }
            
            results = {}
            with self.engine.connect() as conn:
                for name, query in queries.items():
                    df_result = pd.read_sql(query, conn)
                    results[name] = df_result
                    logging.info(f"\n{name}:")
                    logging.info(f"{df_result.to_string()}")
            
            return results
        except Exception as e:
            logging.error(f"Sample analysis failed: {e}")
            return None
    
    def get_database_stats(self):
        """Get database statistics"""
        try:
            stats_queries = {
                "Table Row Counts": """
                    SELECT 
                        'dimension_date' as table_name, COUNT(*) as row_count FROM dimension_date
                    UNION ALL
                    SELECT 'dimension_product', COUNT(*) FROM dimension_product
                    UNION ALL
                    SELECT 'dimension_location', COUNT(*) FROM dimension_location
                    UNION ALL
                    SELECT 'dimension_order', COUNT(*) FROM dimension_order
                    UNION ALL
                    SELECT 'fact_sales', COUNT(*) FROM fact_sales
                """,
                "Database Size": """
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """
            }
            
            with self.engine.connect() as conn:
                for name, query in stats_queries.items():
                    df_result = pd.read_sql(query, conn)
                    logging.info(f"\n{name}:")
                    logging.info(f"{df_result.to_string()}")
            
            return True
        except Exception as e:
            logging.error(f"Failed to get database stats: {e}")
            return False

def main():
    logging.info("Starting Data Warehouse Implementation")
    
    dw = SalesDataWarehouse()
    
    if not dw.test_connection():
        logging.error("Cannot connect to database. Please check your PostgreSQL connection.")
        return
    
    if not dw.create_database_schema():
        logging.error("Failed to create database schema")
        return
    
    try:
        # Folder langsung: "processed_data"
        base_path = os.path.abspath("processed_data")
        
        # Load CSV files
        df_date = pd.read_csv(os.path.join(base_path, "dimension_date.csv"), parse_dates=["order_date"])
        df_product = pd.read_csv(os.path.join(base_path, "dimension_product.csv"))
        df_location = pd.read_csv(os.path.join(base_path, "dimension_location.csv"))
        df_order = pd.read_csv(os.path.join(base_path, "dimension_order.csv"), parse_dates=["order_date"])
        df_fact = pd.read_csv(os.path.join(base_path, "fact_sales.csv"))

        # Insert data
        df_date.to_sql("dimension_date", dw.engine, if_exists="append", index=False, method='multi')
        df_product.to_sql("dimension_product", dw.engine, if_exists="append", index=False, method='multi')
        df_location.to_sql("dimension_location", dw.engine, if_exists="append", index=False, method='multi')
        df_order.to_sql("dimension_order", dw.engine, if_exists="append", index=False, method='multi')
        df_fact.to_sql("fact_sales", dw.engine, if_exists="append", index=False, method='multi')

        # Indexes, views, analysis
        dw.create_indexes()
        dw.create_analysis_views()
        dw.run_sample_analysis()
        dw.get_database_stats()

        logging.info("Data Warehouse implementation completed successfully!")

    except Exception as e:
        logging.error(f"Implementation failed: {e}")


if __name__ == "__main__":
    main()