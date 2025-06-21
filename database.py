import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Optional

class ForumDatabase:
    def __init__(self, db_path: str = "forum.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
        return conn
    
    def init_database(self):
        """初始化数据库表结构"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                avatar TEXT,
                join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                post_count INTEGER DEFAULT 0,
                reputation INTEGER DEFAULT 0
            )
        ''')
        
        # 创建分类表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                post_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建帖子表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                author_id INTEGER NOT NULL,
                category_id INTEGER NOT NULL,
                views INTEGER DEFAULT 0,
                likes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (author_id) REFERENCES users (id),
                FOREIGN KEY (category_id) REFERENCES categories (id)
            )
        ''')
        
        # 创建标签表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                post_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建帖子标签关联表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS post_tags (
                post_id INTEGER NOT NULL,
                tag_id INTEGER NOT NULL,
                PRIMARY KEY (post_id, tag_id),
                FOREIGN KEY (post_id) REFERENCES posts (id),
                FOREIGN KEY (tag_id) REFERENCES tags (id)
            )
        ''')
        
        # 创建回复表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS replies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                author_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                likes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (post_id) REFERENCES posts (id),
                FOREIGN KEY (author_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        # 插入初始数据
        self.insert_initial_data()
    
    def insert_initial_data(self):
        """插入初始数据"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 插入默认用户
        default_users = [
            ('Trading Expert', 'expert@example.com', '👨‍💼'),
            ('Stock Newbie', 'newbie@example.com', '👶'),
            ('Senior Investor', 'investor@example.com', '👴'),
            ('Market Analyst', 'analyst@example.com', '📊'),
            ('Risk Manager', 'risk@example.com', '🛡️')
        ]
        
        for username, email, avatar in default_users:
            cursor.execute('''
                INSERT OR IGNORE INTO users (username, email, avatar)
                VALUES (?, ?, ?)
            ''', (username, email, avatar))
        
        # 插入默认分类
        categories = [
            ('Technical Analysis', 'Technical analysis and chart patterns'),
            ('Fundamental Analysis', 'Fundamental analysis and financial statements'),
            ('Investment Strategy', 'Investment strategies and portfolio management'),
            ('Market News', 'Market news and updates'),
            ('Beginner\'s Guide', 'Beginner-friendly guides and tutorials'),
            ('Experience Sharing', 'Personal investment experiences and stories')
        ]
        
        for name, description in categories:
            cursor.execute('''
                INSERT OR IGNORE INTO categories (name, description)
                VALUES (?, ?)
            ''', (name, description))
        
        # 插入默认标签
        tags = [
            'MACD', 'RSI', 'Bollinger Bands', 'Technical Analysis', 'Fundamentals',
            'P/E Ratio', 'Financial Analysis', 'Beginner', 'Value Investing',
            'Stock Selection', 'Investment Strategy', 'Risk Management',
            'Stop Loss', 'Trading', 'Sector Analysis', 'Market Trends'
        ]
        
        for tag_name in tags:
            cursor.execute('''
                INSERT OR IGNORE INTO tags (name)
                VALUES (?)
            ''', (tag_name,))
        
        # 插入示例帖子
        sample_posts = [
            {
                'title': 'How to Analyze Technical Indicators: MACD, RSI, and Bollinger Bands',
                'content': 'Technical analysis is a powerful tool for stock trading. In this post, I\'ll explain how to use MACD, RSI, and Bollinger Bands effectively.\n\nMACD (Moving Average Convergence Divergence) helps identify trend changes and momentum.\nRSI (Relative Strength Index) measures overbought and oversold conditions.\nBollinger Bands show price volatility and potential reversal points.',
                'author': 'Trading Expert',
                'category': 'Technical Analysis',
                'tags': ['Technical Analysis', 'MACD', 'RSI', 'Bollinger Bands'],
                'views': 156,
                'likes': 23
            },
            {
                'title': 'Beginner\'s Guide: Understanding P/E Ratio and Financial Statements',
                'content': 'As a beginner investor, understanding financial ratios is crucial. The P/E ratio is one of the most important metrics to evaluate a stock.\n\nP/E Ratio = Price per Share / Earnings per Share\n\nA lower P/E ratio might indicate undervaluation, while a higher P/E might suggest overvaluation.',
                'author': 'Stock Newbie',
                'category': 'Fundamental Analysis',
                'tags': ['Fundamentals', 'P/E Ratio', 'Financial Analysis', 'Beginner'],
                'views': 98,
                'likes': 15
            },
            {
                'title': 'My Value Investing Strategy: Finding Undervalued Stocks',
                'content': 'Value investing has been my approach for over 10 years. Here\'s my strategy for finding undervalued stocks:\n\n1. Look for companies with strong fundamentals\n2. Check for low P/E ratios compared to industry average\n3. Analyze debt levels and cash flow\n4. Consider the company\'s competitive advantage',
                'author': 'Senior Investor',
                'category': 'Investment Strategy',
                'tags': ['Value Investing', 'Stock Selection', 'Investment Strategy'],
                'views': 245,
                'likes': 31
            }
        ]
        
        for post_data in sample_posts:
            # 获取作者ID
            cursor.execute('SELECT id FROM users WHERE username = ?', (post_data['author'],))
            author_id = cursor.fetchone()['id']
            
            # 获取分类ID
            cursor.execute('SELECT id FROM categories WHERE name = ?', (post_data['category'],))
            category_id = cursor.fetchone()['id']
            
            # 插入帖子
            cursor.execute('''
                INSERT INTO posts (title, content, author_id, category_id, views, likes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (post_data['title'], post_data['content'], author_id, category_id, 
                  post_data['views'], post_data['likes']))
            
            post_id = cursor.lastrowid
            
            # 插入标签关联
            for tag_name in post_data['tags']:
                cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
                tag_result = cursor.fetchone()
                if tag_result:
                    tag_id = tag_result['id']
                    cursor.execute('''
                        INSERT OR IGNORE INTO post_tags (post_id, tag_id)
                        VALUES (?, ?)
                    ''', (post_id, tag_id))
        
        conn.commit()
        conn.close()
    
    def get_all_posts(self) -> List[Dict]:
        """获取所有帖子"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                p.id, p.title, p.content, p.views, p.likes, p.created_at,
                u.username as author, u.avatar,
                c.name as category,
                GROUP_CONCAT(t.name) as tags
            FROM posts p
            JOIN users u ON p.author_id = u.id
            JOIN categories c ON p.category_id = c.id
            LEFT JOIN post_tags pt ON p.id = pt.post_id
            LEFT JOIN tags t ON pt.tag_id = t.id
            GROUP BY p.id
            ORDER BY p.created_at DESC
        ''')
        
        posts = []
        for row in cursor.fetchall():
            post = dict(row)
            post['tags'] = post['tags'].split(',') if post['tags'] else []
            posts.append(post)
        
        conn.close()
        return posts
    
    def get_categories(self) -> List[Dict]:
        """获取所有分类"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, description, post_count
            FROM categories
            ORDER BY post_count DESC
        ''')
        
        categories = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return categories
    
    def get_tags(self) -> List[Dict]:
        """获取所有标签"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, post_count
            FROM tags
            ORDER BY post_count DESC
            LIMIT 20
        ''')
        
        tags = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return tags
    
    def create_post(self, title: str, content: str, author: str, category: str, tags: List[str]) -> int:
        """创建新帖子"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 获取或创建作者
            cursor.execute('SELECT id FROM users WHERE username = ?', (author,))
            user_result = cursor.fetchone()
            if user_result:
                author_id = user_result['id']
            else:
                cursor.execute('INSERT INTO users (username) VALUES (?)', (author,))
                author_id = cursor.lastrowid
            
            # 获取分类ID
            cursor.execute('SELECT id FROM categories WHERE name = ?', (category,))
            category_result = cursor.fetchone()
            if not category_result:
                raise ValueError(f"Category '{category}' not found")
            category_id = category_result['id']
            
            # 插入帖子
            cursor.execute('''
                INSERT INTO posts (title, content, author_id, category_id)
                VALUES (?, ?, ?, ?)
            ''', (title, content, author_id, category_id))
            
            post_id = cursor.lastrowid
            
            # 处理标签
            for tag_name in tags:
                # 获取或创建标签
                cursor.execute('SELECT id FROM tags WHERE name = ?', (tag_name,))
                tag_result = cursor.fetchone()
                if tag_result:
                    tag_id = tag_result['id']
                else:
                    cursor.execute('INSERT INTO tags (name) VALUES (?)', (tag_name,))
                    tag_id = cursor.lastrowid
                
                # 关联帖子和标签
                cursor.execute('''
                    INSERT OR IGNORE INTO post_tags (post_id, tag_id)
                    VALUES (?, ?)
                ''', (post_id, tag_id))
            
            # 更新分类帖子计数
            cursor.execute('''
                UPDATE categories 
                SET post_count = (
                    SELECT COUNT(*) FROM posts WHERE category_id = ?
                )
                WHERE id = ?
            ''', (category_id, category_id))
            
            # 更新标签帖子计数
            for tag_name in tags:
                cursor.execute('''
                    UPDATE tags 
                    SET post_count = (
                        SELECT COUNT(*) FROM post_tags pt 
                        JOIN posts p ON pt.post_id = p.id 
                        WHERE pt.tag_id = tags.id
                    )
                    WHERE name = ?
                ''', (tag_name,))
            
            conn.commit()
            return post_id
            
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def get_forum_stats(self) -> Dict:
        """获取论坛统计信息"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as total_posts FROM posts')
        total_posts = cursor.fetchone()['total_posts']
        
        cursor.execute('SELECT COUNT(*) as total_users FROM users')
        total_users = cursor.fetchone()['total_users']
        
        cursor.execute('''
            SELECT COUNT(*) as posts_today 
            FROM posts 
            WHERE DATE(created_at) = DATE('now')
        ''')
        posts_today = cursor.fetchone()['posts_today']
        
        cursor.execute('SELECT COUNT(*) as total_categories FROM categories')
        total_categories = cursor.fetchone()['total_categories']
        
        conn.close()
        
        return {
            'total_posts': total_posts,
            'total_users': total_users,
            'posts_today': posts_today,
            'total_categories': total_categories
        }

# 创建全局数据库实例
db = ForumDatabase() 