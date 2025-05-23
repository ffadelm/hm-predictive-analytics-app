-- SQL Script untuk membuat skema data warehouse
-- Generated pada: 2025-05-20 14:18:29

-- Create dimension_date table
CREATE TABLE dimension_date (
    date_id INTEGER PRIMARY KEY,
    order_date DATE,
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
    location_id INTEGER PRIMARY KEY,
    country VARCHAR(50),
    city VARCHAR(100),
    state VARCHAR(50),
    region VARCHAR(20)
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

