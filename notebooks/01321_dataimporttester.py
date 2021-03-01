from data.read_binary import readDNSdata
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# quantities, quanList, xF, yF, zF, length, storl, paraString = readDNSdata('../data/raw/field.0493.u', onlyU=False)


# Finde de forskellige filer med os
b = os.listdir("../data/raw")

# TODO Med listen over de forskellige filer skal der nu laves en funktion som loader hver enkelt datasæt ind og
#   gemmer dem som h5py filer i intermediate mappen. GØRES VIA DASK


# TODO De filer skal jeg så i et nyt script loades ind i et nyt script, hvor der for hver enkelt regnes statestik(endnu et script) ELLER
#   Samler dem i forskellige batchstørrelser og så regner det derefter? OGSÅ I DASK
