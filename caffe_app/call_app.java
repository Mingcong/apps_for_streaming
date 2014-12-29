public class call_app {
         static
         {
                   System.loadLibrary("caffe_app");
         }
         public native static int app(int batch_size, String mode, String input, String output);
         public static void main(String[] args)
         {
                   System.out.println(System.getProperty("java.library.path"));
		   int batch_size = 500;
		   String mode = "CPU";
		   String input = "0";
		   String output = "out";
                   call_app.app(batch_size, mode, input, output);
         }
}
