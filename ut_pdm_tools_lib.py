from datetime import datetime, timedelta

class CommonFunctions:
    
    def read_from_file(filename):
        f = open(filename, 'r')
        content = ""
        for row in f:
            content += row
        return content.split("\n")
    
    def mark_timestamp():
        return datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")