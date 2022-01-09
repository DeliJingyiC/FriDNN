# If database file doesn't exist, download from webdav.coredumped.tech: https://webdav.coredumped.tech/s/5r4JD97FpyjmYtj
if [ ! -f "timit_LDC93S1.zip" ]; then
    wget https://webdav.coredumped.tech/s/gcAjggrzeR2msYe/download/timit_LDC93S1.zip
fi

# If database folder doesn't exist, unzip database file
if [ ! -d "timit" ]; then
    unzip timit_LDC93S1.zip
fi