import clang.cindex
LIBCLANG_PATH = '/Library/Developer/CommandLineTools/usr/lib/libclang.dylib'
# create an index
clang.cindex.Config.set_library_file(LIBCLANG_PATH)

def list_functions(tu):
    """
    Print the names of all the functions in the translation unit.
    """
    filename = tu.cursor.spelling
    for cursor in tu.cursor.walk_preorder():
        if cursor.location.file is None:
            continue
        if cursor.location.file.name != filename:
            continue
        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            print(cursor.spelling)

if __name__ == '__main__':
    index = clang.cindex.Index.create()
    tu = index.parse('test.cpp')  # Replace 'sample.cpp' with your file
    list_functions(tu)
    #print(tu.cursor.spelling)
    #print(list_functions(tu))

