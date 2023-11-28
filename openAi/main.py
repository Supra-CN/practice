# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import logging
import os
import openai
import json

print(f"openai api_key = {os.environ.get('OPENAI_API_KEY')}")
print(f"openai organization = {os.environ.get('OPENAI_ORGANIZATION')}")

def create_image():
    openai.Image.create()

def print_model_list():
    print(f"openai.Model.list() start")
    model_list = openai.Model.list()
    print(f"openai.Model.list() start end modelList: {model_list}")
    with open("output/modelList.json", "w", encoding='utf-8') as out_file:
        json.dump(model_list, out_file, indent=4)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # print_model_list()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
