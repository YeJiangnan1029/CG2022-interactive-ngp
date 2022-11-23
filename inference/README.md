inference.py定义模型文件调用的接口

方法：

​	***NerfRunner.inference***

参数：
    ```
    pos: [float: x, float: y, float: z] 
    	场景相机的坐标 
    dir: [float: x_angle, float: y_angle,float:  z_angle] 
    	场景相机的方向 用欧拉角表示 单位是度 
    euler_mode: str
    	欧拉角模式，指定绕三个轴旋转的顺序
    ```

返回：

```
 (H, W, 3) np.uint8 RGB通道
 	推理出的图片
```

 