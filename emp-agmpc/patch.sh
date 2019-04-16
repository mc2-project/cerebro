pushd emp-tool/
git apply ../netio.patch
make -j16
sudo make install
popd
make -j16
