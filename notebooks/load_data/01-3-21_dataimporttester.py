from data.read_binary import readDNSdata
from dotenv import load_dotenv, find_dotenv
import os
from memory_profiler import profile

quantities, quanList, xF, yF, zF, length, storl, paraString=readDNSdata('../data/raw/field.0495.u')