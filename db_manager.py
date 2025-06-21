#!/usr/bin/env python3
"""
论坛数据库管理工具
用于查看和管理论坛数据
"""

import sqlite3
from database import db
import json
from datetime import datetime

def print_separator(title):
    """打印分隔符"""
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50)

def view_all_posts():
    """查看所有帖子"""
    print_separator("所有帖子")
    posts = db.get_all_posts()
    
    if not posts:
        print("暂无帖子")
        return
    
    for i, post in enumerate(posts, 1):
        print(f"\n{i}. {post['title']}")
        print(f"   作者: {post['author']}")
        print(f"   分类: {post['category']}")
        print(f"   浏览量: {post['views']}, 点赞: {post['likes']}")
        print(f"   标签: {', '.join(post['tags'])}")
        print(f"   发布时间: {post['created_at']}")
        print(f"   内容预览: {post['content'][:100]}...")

def view_categories():
    """查看所有分类"""
    print_separator("所有分类")
    categories = db.get_categories()
    
    if not categories:
        print("暂无分类")
        return
    
    for category in categories:
        print(f"- {category['name']} ({category['post_count']} 个帖子)")
        if category['description']:
            print(f"  描述: {category['description']}")

def view_tags():
    """查看所有标签"""
    print_separator("热门标签")
    tags = db.get_tags()
    
    if not tags:
        print("暂无标签")
        return
    
    for tag in tags:
        print(f"- {tag['name']} ({tag['post_count']} 个帖子)")

def view_stats():
    """查看论坛统计"""
    print_separator("论坛统计")
    stats = db.get_forum_stats()
    
    print(f"总帖子数: {stats['total_posts']}")
    print(f"总用户数: {stats['total_users']}")
    print(f"今日帖子: {stats['posts_today']}")
    print(f"分类数量: {stats['total_categories']}")

def create_sample_post():
    """创建示例帖子"""
    print_separator("创建示例帖子")
    
    try:
        post_id = db.create_post(
            title="示例帖子：数据库集成测试",
            content="这是一个测试帖子，用于验证数据库功能是否正常工作。\n\n包含多行内容测试。",
            author="System Test",
            category="Technical Analysis",
            tags=["测试", "数据库", "集成"]
        )
        print(f"示例帖子创建成功，ID: {post_id}")
    except Exception as e:
        print(f"创建示例帖子失败: {e}")

def reset_database():
    """重置数据库"""
    print_separator("重置数据库")
    confirm = input("确定要重置数据库吗？这将删除所有数据！(y/N): ")
    
    if confirm.lower() == 'y':
        try:
            # 删除数据库文件
            import os
            if os.path.exists("forum.db"):
                os.remove("forum.db")
                print("数据库文件已删除")
            
            # 重新初始化数据库
            from database import ForumDatabase
            new_db = ForumDatabase()
            print("数据库已重新初始化")
        except Exception as e:
            print(f"重置数据库失败: {e}")
    else:
        print("操作已取消")

def main():
    """主菜单"""
    while True:
        print_separator("论坛数据库管理工具")
        print("1. 查看所有帖子")
        print("2. 查看所有分类")
        print("3. 查看热门标签")
        print("4. 查看论坛统计")
        print("5. 创建示例帖子")
        print("6. 重置数据库")
        print("0. 退出")
        
        choice = input("\n请选择操作 (0-6): ").strip()
        
        if choice == '1':
            view_all_posts()
        elif choice == '2':
            view_categories()
        elif choice == '3':
            view_tags()
        elif choice == '4':
            view_stats()
        elif choice == '5':
            create_sample_post()
        elif choice == '6':
            reset_database()
        elif choice == '0':
            print("再见！")
            break
        else:
            print("无效选择，请重试")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main() 