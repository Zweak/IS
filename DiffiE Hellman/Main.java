import java.util.*;
public class Main
{
public static void main(String args[])
{
Scanner sc=new Scanner(System.in);
System.out.println("Enter modulo(n)");
long n=sc.nextInt();
System.out.println("Enter primitive root of "+n);
long g=sc.nextInt();
System.out.println("Choose 1st secret no(Alice):X_a:");
long X_a=sc.nextInt();
System.out.println("Choose 2nd secret no(BOB):X_b");
long X_b=sc.nextInt();
long Y_a = (long)Math.pow(g,X_a)%n;
long Y_b = (long)Math.pow(g,X_b)%n;
long K_A = (long)Math.pow(Y_b,X_a)%n;
System.out.println("They share a secret Key for User-A is = "+K_A);
long K_B = (long)Math.pow(Y_a,X_b)%n;
System.out.println("They share a secret Key for User-B is = "+K_B);
if(K_A==K_B)
{
System.out.println("ALice and Bob can communicate with each other!!!");
System.out.println("They share a secret no = "+K_B);
}
else
{
    System.out.println("ALice and Bob cannot communicate with each other!!!");
}
}
}
