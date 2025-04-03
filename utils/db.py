import mysql.connector
from mysql.connector import Error

def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',           # Replace with your MySQL username
            password='Pass@123',           # Replace with your MySQL password
            database='blood_group_prediction'
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def initialize_db():
    try:
        # First connect without specifying a database
        connection = mysql.connector.connect(
            host='localhost',
            user='root',           # Replace with your MySQL username
            password='Pass@123'            # Replace with your MySQL password
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # Create database if it doesn't exist
            cursor.execute("CREATE DATABASE IF NOT EXISTS blood_group_prediction")
            cursor.execute("USE blood_group_prediction")
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    email VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    age INT NOT NULL,
                    weight FLOAT,
                    gender VARCHAR(10),
                    address TEXT,
                    phone VARCHAR(20),
                    blood_group VARCHAR(10),
                    fingerprint_path VARCHAR(255),
                    user_id INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            print("Database and tables created successfully")
            cursor.close()
            connection.close()
            
    except Error as e:
        print(f"Error initializing database: {e}")
