# import pprint
# from datetime import datetime, timedelta

# import pyarrow.parquet as pq
# import pandas as pd
# import akshare as ak
# import pickle


# def func_foreign():
#     all_stocks = ak.stock_zh_a_spot_em()
#     # 保存数据到 CSV 文件
#     csv_path = 'all_stocks.csv'
#     all_stocks.to_csv(csv_path, index=False)
#     print(f"数据已保存到 {csv_path}")
#     pprint.pprint(all_stocks)


# def func_domestic():
#     domestic_stocks = ak.stock_zh_a_spot_em()
#     # 获取美股
#     foreign_stock = ak.stock_us_spot_em()
#     all_stocks = pd.concat([domestic_stocks, foreign_stock], axis=0, ignore_index=True)


# def func():
#     # url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=AAPL&apikey=4GDA0WD6VJE8YBHY&datatype=csv'
#     # r = requests.get(url).content
#     # data = pd.read_csv(io.StringIO(r.decode('utf-8')))
#     # print(data)
#     # foreign_stock = ak.stock_zh_a_spot_em()
#     # pprint(foreign_stock)
#     end_date = datetime.now().strftime('%Y%m%d')
#     start_date = (datetime.strptime(end_date, '%Y%m%d') -
#                   timedelta(30)).strftime('%Y%m%d')
#     hist_data = ak.stock_zh_a_hist(symbol="600340",
#                                    start_date=start_date,
#                                    end_date=end_date)
#     print(hist_data)


# def serialize_dataframe(df, file_path):
#     try:
#         with open(file_path, 'wb') as f:
#             pickle.dump(df, f)
#         print(f"DataFrame 已成功序列化到 {file_path}")
#     except Exception as e:
#         print(f"序列化时出现错误: {e}")


# def deserialize_dataframe(file_path):
#     try:
#         with open(file_path, 'rb') as f:
#             df = pickle.load(f)
#         print(f"DataFrame 已成功从 {file_path} 反序列化")
#         return df
#     except Exception as e:
#         print(f"反序列化时出现错误: {e}")
#         return None



# def fun2():
#     # domestic_stocks = ak.stock_zh_a_spot_em()
#     # foreign_stock = ak.stock_us_spot_em()
#     # all_stocks = pd.concat([domestic_stocks, foreign_stock], axis=0, ignore_index=True)
#     #
#     # # 示例使用
#     file_path = 'foreign_stocks.pkl'
#     # serialize_dataframe(foreign_stock, file_path)
#     deserialized_df = deserialize_dataframe(file_path)
#     if deserialized_df is not None:
#         print("反序列化后的 DataFrame 基本信息：")
#         deserialized_df.info()

# def read_parquet():
#     table_read = pq.read_table('data_cache/20250404_20250504/300005_hist.parquet')
#     df_read = table_read.to_pandas()
#     print(len(df_read))
#     print(df_read)


# if __name__ == "__main__":
#     fun2()

import pprint
from datetime import datetime, timedelta

import pyarrow.parquet as pq
import pandas as pd
import akshare as ak
import pickle


def func_foreign():
    all_stocks = ak.stock_zh_a_spot_em()
    # 保存数据到 CSV 文件
    csv_path = 'all_stocks.csv'
    all_stocks.to_csv(csv_path, index=False)
    print(f"数据已保存到 {csv_path}")
    pprint.pprint(all_stocks)


def func_domestic():
    domestic_stocks = ak.stock_zh_a_spot_em()
    # 获取美股
    foreign_stock = ak.stock_us_spot_em()
    all_stocks = pd.concat([domestic_stocks, foreign_stock], axis=0, ignore_index=True)


def func():
    # url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=AAPL&apikey=4GDA0WD6VJE8YBHY&datatype=csv'
    # r = requests.get(url).content
    # data = pd.read_csv(io.StringIO(r.decode('utf-8')))
    # print(data)
    # foreign_stock = ak.stock_zh_a_spot_em()
    # pprint(foreign_stock)
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.strptime(end_date, '%Y%m%d') -
                  timedelta(30)).strftime('%Y%m%d')
    hist_data = ak.stock_zh_a_hist(symbol="600340",
                                   start_date=start_date,
                                   end_date=end_date)
    print(hist_data)


def serialize_dataframe(df, file_path):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"DataFrame 已成功序列化到 {file_path}")
    except Exception as e:
        print(f"序列化时出现错误: {e}")


def deserialize_dataframe(file_path):
    try:
        with open(file_path, 'rb') as f:
            df = pickle.load(f)
        print(f"DataFrame 已成功从 {file_path} 反序列化")
        return df
    except Exception as e:
        print(f"反序列化时出现错误: {e}")
        return None



def fun2():
    domestic_stocks = ak.stock_zh_a_spot_em()
    foreign_stock = ak.stock_us_spot_em()
    all_stocks = pd.concat([domestic_stocks, foreign_stock], axis=0, ignore_index=True)
    # # 示例使用
    file_path = 'all_stocks.pkl'
    serialize_dataframe(all_stocks, file_path)
    # deserialized_df = d

if __name__ == "__main__":
    fun2()