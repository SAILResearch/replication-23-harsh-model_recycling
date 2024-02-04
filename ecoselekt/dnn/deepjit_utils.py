import pickle

import numpy as np
from tqdm import tqdm

from ecoselekt.settings import settings


def padding_commit_code_line(data, max_line, max_length):
    new_data = list()
    for d in data:
        if len(d) == max_line:
            new_data.append(d)
        elif len(d) > max_line:
            new_data.append(d[:max_line])
        else:
            num_added_line = max_line - len(d)
            for i in range(num_added_line):
                d.append(("<NULL> " * max_length).strip())
            new_data.append(d)
    return new_data


def padding_multiple_length(lines, max_length):
    return [padding_length(line=l, max_length=max_length) for l in lines]


def padding_length(line, max_length):
    line_length = len(line.split())
    if line_length < max_length:
        return str(line + " <NULL>" * (max_length - line_length)).strip()
    elif line_length > max_length:
        line_split = line.split()
        return " ".join([line_split[i] for i in range(max_length)])
    else:
        return line


def padding_data(data, dictionary, params, type):
    if type == "msg":
        pad_msg = padding_message(data=data, max_length=params["msg_length"])
        pad_msg = mapping_dict_msg(pad_msg=pad_msg, dict_msg=dictionary)
        return pad_msg
    elif type == "code":
        pad_code = padding_commit_code(
            data=data, max_line=params["code_line"], max_length=params["code_length"]
        )
        pad_code = mapping_dict_code(pad_code=pad_code, dict_code=dictionary)
        return pad_code
    else:
        print("Your type is incorrect -- please correct it")
        exit()


def padding_message(data, max_length):
    new_data = list()
    for d in data:
        new_data.append(padding_length(line=d, max_length=max_length))
    return new_data


def mapping_dict_msg(pad_msg, dict_msg):
    return np.array(
        [
            np.array(
                [
                    dict_msg[w.lower()] if w.lower() in dict_msg.keys() else dict_msg["<NULL>"]
                    for w in line.split()
                ]
            )
            for line in pad_msg
        ]
    )


def mapping_dict_code(pad_code, dict_code):
    new_pad = [
        np.array(
            [
                np.array(
                    [
                        dict_code[w.lower()] if w.lower() in dict_code else dict_code["<NULL>"]
                        for w in l.split()
                    ]
                )
                for l in ml
            ]
        )
        for ml in pad_code
    ]
    return np.array(new_pad)


def padding_commit_code(data, max_line, max_length):
    padding_length = padding_commit_code_length(data=data, max_length=max_length)
    print(f"Shape of padding_length: {len(padding_length)}")
    padding_line = padding_commit_code_line(
        padding_length, max_line=max_line, max_length=max_length
    )
    print(f"Shape of padding_line: {len(padding_line)}")
    return padding_line


def padding_commit_code_length(data, max_length):
    return [padding_multiple_length(lines=commit, max_length=max_length) for commit in data]


def optim_padding_code(project_name, i, data, dict_code):
    # optimized version of padding_data for code
    max_line = settings.CODE_LINE
    max_length = settings.CODE_LENGTH

    pad_code = []
    for lines in tqdm(data, desc="Processing lines"):
        processed_lines = []
        for line in lines:
            line_length = len(line.split())
            if line_length < max_length:
                line += " <NULL>" * (max_length - line_length)
            elif line_length > max_length:
                line = " ".join(line.split()[:max_length])

            processed_lines.append(line.strip())

        # Pad or truncate lines
        num_added_lines = max(0, max_line - len(processed_lines))
        processed_lines += [("<NULL> " * max_length).strip()] * num_added_lines
        processed_lines = processed_lines[:max_line]

        # Convert words to indices using the dictionary
        pad_code.append(
            [
                [dict_code.get(w.lower(), dict_code["<NULL>"]) for w in l.split()]
                for l in processed_lines
            ]
        )
    return np.array(pad_code)


def optim_padding_msg(project_name, i, data, dict_msg):
    # optimized version of padding_data for msg
    max_length = settings.MSG_LENGTH
    pad_msg = []
    for msg in tqdm(data, desc="Processing messages"):
        line_length = len(msg.split())
        if line_length < max_length:
            msg += " <NULL>" * (max_length - line_length)
        elif line_length > max_length:
            msg = " ".join(msg.split()[:max_length])
        pad_msg.append(msg.strip())

    # Convert words to indices using the dictionary
    pad_msg = np.array(
        [
            np.array([dict_msg.get(w.lower(), dict_msg["<NULL>"]) for w in line.split()])
            for line in pad_msg
        ]
    )
    return pad_msg
