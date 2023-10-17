def recursive_pretty_print(data, name="", max_len=5, level=0):
    if isinstance(data, list):
        print(f"{'    ' * (level)}{name}: {len(data)} elements")
        for i, item in enumerate(data):
            if isinstance(item, (list, dict)):
                recursive_pretty_print(item, f"{name}[{i}]", max_len, level + 1)
            else:
                if isinstance(item, str) and len(item) > max_len:
                    item = f'"{item[:max_len]}..." (and {len(item) - max_len} more chars)'
                print(f"{'    ' * (level)}{name}[{i}]: {item}")

    elif isinstance(data, dict):
        print(f"{name}: {len(data)} elements")
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                recursive_pretty_print(value, f"{name}['{key}']", max_len, level + 1)
            else:
                if isinstance(value, str) and len(value) > max_len:
                    value = f'"{value[:max_len]}..." (and {len(value) - max_len} more chars)'
                print(f"{'    ' * (level)}{name}['{key}']: {value}")
    else:
        print(f"{name}: {data}")

if __name__=="__main__":

    x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, [1, [[[[[[[[2]]]]]]]]], 33, "a really long string"]]
    recursive_pretty_print(x, "x", max_len=16)
