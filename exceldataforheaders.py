import pandas as pd

def process_excel_headers(excel_file, sheet_name):
  """
  Обрабатывает заголовки столбцов в Excel-файле.

  Args:
      excel_file (str): Путь к Excel-файлу.
      sheet_name (str): Название листа в Excel-файле.

  Returns:
      tuple: Кортеж из трех списков:
          - Исходные заголовки столбцов.
          - Заголовки столбцов без первого элемента.
          - Список обновленных заголовков с префиксом и суффиксом "ssx".
  """
  df = pd.read_excel(excel_file, sheet_name=sheet_name)

  fileheaders = list(df.columns)
  fileheaders  = [header.lower() for header in fileheaders]
  headerskey = fileheaders[1:]
  updated_list = ["ssx" + item for item in headerskey]
  jsonstring = str(updated_list)
  jsonstring = jsonstring.replace("'", '"')    
  return fileheaders, headerskey, updated_list, jsonstring