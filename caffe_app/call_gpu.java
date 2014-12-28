public class call_gpu {
         static
         {
                   System.loadLibrary("gpu_app");
         }
         public native static int app(int batch_size, String mode, String input, String output);
         public static void main(String[] args)
         {
                   System.out.println(System.getProperty("java.library.path"));
		   int batch_size = 500;
		   String mode = "GPU";
		   String input = "0";
		   String output = "out";
                   call_gpu.app(batch_size, mode, input, output);
         }
}
