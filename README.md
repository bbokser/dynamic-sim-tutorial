# dynamic-sim-tutorial

## Converting Images to Video

In a separate terminal:

```shell
cd imgs/ 
```
The first time you run this:

```shell
cat *.png | ffmpeg -f image2pipe -i - ../out.avi && ffmpeg -i ../out.avi -pix_fmt rgb24 -loop 0 ../out.gif 
```

In future iterations, this helps to remove the previous files:
```shell
rm ../out.avi ../out.gif && cat *.png | ffmpeg -f image2pipe -i - ../out.avi && ffmpeg -i ../out.avi -pix_fmt rgb24 -loop 0 ../out.gif 
```