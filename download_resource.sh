move_dir_to_shell_file() {
    dir_shell_file=`dirname "$0"`
    cd ${dir_shell_file}
}

download_and_extract() {
    local url=$1
    echo "Downloading ${url}"
    local ext=${url##*.}
    if [ `echo ${ext} | grep zip` ]; then
        curl -Lo temp.zip ${url}
        unzip -o temp.zip
        rm temp.zip
    else
        curl -Lo temp.tgz ${url}
        tar xzvf temp.tgz
        rm temp.tgz
    fi
}
########################################################################

move_dir_to_shell_file
download_and_extract "https://github.com/iwatake2222/self-driving-ish_computer_vision_system/releases/download/20210914/resource.zip"
