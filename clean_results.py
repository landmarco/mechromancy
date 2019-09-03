import os

cwd = os.getcwd()
results_dir = os.path.join(cwd, 'results_2t')
def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
experiments = get_subdirectories(results_dir)
num_epochs = 5
hit_list = ["params_e"+str(n)+"_" for n in xrange(num_epochs)]
unused_files = []
for e in experiments:
    e_dir = os.path.join(results_dir, e)
    params = os.path.join(e_dir, "params")
    for root, dirs, files in os.walk(params):
        for file in files:
            for hit in hit_list:
                if file.startswith(hit):
                    print file
                    unused_files.append(os.path.join(root, file))
def prompt_delete(num_prompts):
    num_prompts -= 1
    if num_prompts >= 0:
        prompt = input("Do you want to delete these "+str(len(unused_files))+" files? ['Y'/'n']")
        if prompt == "Y" or prompt == "yes":
            print 'removing old epochs...'
            for uf in unused_files:
                os.remove(uf)
        elif prompt == "n" or prompt == "no":
            print "clean aborted: 0 files deleted"
        else:
            print "warning:", prompt, "is an unknown command"
            prompt_delete(num_prompts)
    else:
        print "0 files deleted: Good-bye"
if len(unused_files) > 0:
    prompt_delete(3)
else:
    print 'found 0 files to clean: Good-bye'
