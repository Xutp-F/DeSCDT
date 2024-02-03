# ---------------------------------
# --------人生苦短，我用python--------
# ---------------------------------
# -*- coding: UTF-8 -*-
import random
import os
import sys
import re
# from interval import Interval
import argparse
from random import choice




def mutators_strategy(text, mode,):
    print(text)

    if mode == "m1":
        if text.find("for") >=  0:
            index = text.find("for")
            if index_start == 0:
                prefix = text[:index_start]
                middle = text[index_start:index_end+1]
                tail = text[index_end+1:]
                temp = middle.split(";")
                try:
                    temp[1] += "+99"
                    middle = temp[0] + ";" + temp[1] + ";" + temp[2]
                except IndexError:
                    return  text
                retext = prefix + middle + tail
                return  retext
            else:
                return  text
        else:
            return text

    if mode == "m2":
        comparePredicate = [">=", "<=", ">", "<", "==", "!="]
        if text.find(">=") >= 0:
            while (1):
                temp = choice(comparePredicate)
                if (temp != ">="):
                    break
            text = text.replace(">=", temp)
        if text.find("<=") >= 0:
            while (1):
                temp = choice(comparePredicate)
                if (temp != "<="):
                    break
            text = text.replace("<=", temp)


        if text.find(">") >= 0 and text.find(">=") < 0 and text.find("=>") < 0:
            while (1):
                temp = choice(comparePredicate)
                if (temp != ">"):
                    break
            text = text.replace(">", temp)

        if text.find("<") >= 0 and text.find("<=") < 0:
            while (1):
                temp = choice(comparePredicate)
                if (temp != "<"):
                    break
            text = text.replace("<", temp)
        if text.find("==") >= 0:
            while (1):
                temp = choice(comparePredicate)
                if (temp != "=="):
                    break
            text = text.replace("==", temp)
        if text.find("!=") >= 0:
            while (1):
                temp = choice(comparePredicate)
                if (temp != "!="):
                    break
            text = text.replace("!=", temp)
        return  text




    if mode == "m3":
        comparePredicate = ["+", "-", "*", "/", "%"]
        if text.find("+") >= 0:
            while (1):
                temp = choice(comparePredicate)
                if (temp != "+"):
                    break
            text = text.replace("+", temp)

        if text.find("-") >= 0:
            while (1):
                temp = choice(comparePredicate)
                if (temp != "-"):
                    break
            text = text.replace("-", temp)

        if text.find("/") >= 0 :
            while (1):
                temp = choice(comparePredicate)
                if (temp != "/"):
                    break
            text = text.replace("/", temp)
        return  text

    if mode == "m4":
        comparePredicate = ["&&","!","||"]
        if text.find("||") >= 0:
           text = text.replace( "||", choice(comparePredicate))
        elif text.find("&&") >= 0:
            text = text.replace("&&", choice(comparePredicate))
        elif text.find("!") >= 0:
            text = text.replace("!", choice(comparePredicate))
        return text

    if mode == "m5":
        comparePredicate = ["uint", "uint8", "uint16","int", "int8", "int16","bytes"]
        if text.find("uint") >= 0:
            text = text.replace("uint", choice(comparePredicate))
        elif text.find("uint8") >= 0:
            text = text.replace("uint8", choice(comparePredicate))
        elif text.find("uint16") >= 0:
            text = text.replace("uint16", choice(comparePredicate))
        elif text.find("int") >= 0:
            text = text.replace("int", choice(comparePredicate))
        elif text.find("int8") >= 0:
            text = text.replace("int8", choice(comparePredicate))
        elif text.find("int16") >= 0:
            text = text.replace("int16", choice(comparePredicate))
        elif text.find("bytes") >= 0:
            text = text.replace("bytes", choice(comparePredicate))
        return text