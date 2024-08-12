import os

def compare_text_files(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    if len(lines1) != len(lines2):
        return False

    for line1, line2 in zip(lines1, lines2):
        if line1 != line2:
            return False

    return True

if __name__ == "__main__":
    # Tests: symnmf.py and symnmf.c
    for target in["iris", "spheres","drive1", "drive2"]:
        for goal in ["sym", "norm", "ddg", "symnmf"]:
            path1 = os.path.join("expected-outputs", f"{goal}_{target}.txt")
            path2 = os.path.join("res", f"{goal}_{target}_res.txt")
            print(f"{target} {goal}: {compare_text_files(path1, path2)}")
            if(goal!="symnmf"):
                path2 = os.path.join("res", f"{goal}c_{target}_res.txt")
                print(f"{target} {goal}c: {compare_text_files(path1, path2)}")

    # Tests: analysis.py
    expected_output_directory_in_str = os.path.join("expected-outputs", "analysis")
    res_directory_in_str = os.path.join("res", "analysis")

    res_files_directory = os.fsencode(res_directory_in_str)

    for file in sorted(os.listdir(res_files_directory)):
        res_filename = os.fsdecode(file)
        res_file_path = os.path.join(res_directory_in_str, res_filename)
        expected_outcome_file_path = os.path.join(expected_output_directory_in_str, res_filename)
        if res_filename.endswith(".txt"):
            print(f"{res_filename} analysis: {compare_text_files(res_file_path, expected_outcome_file_path)}")
