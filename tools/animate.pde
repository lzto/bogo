/*
 * Animate bndldx bndstx request
 * parameters: filename, total_cnt, total_bins and max_bin_size
 * 2017 Tong Zhang<ztong@vt.edu>
 */

import java.io.File;
import java.util.Scanner;
import java.util.Queue;
import java.util.LinkedList;

int animate;
int[] bins;
int total_bins;
int max_bin_size;
int total_cnt;

String filename="praw.csv";

Scanner scanner;

int getdata()
{
  if(scanner.hasNextInt())
    return scanner.nextInt();
  return 0;
}

void setup()
{
    try{
      scanner = new Scanner(new File(filename));
      total_cnt = 10000000;//total rows in csv file
      total_bins = 2;//total number of bins
      max_bin_size = 5802138;//the bin with the maximum number of elements
    }catch(Exception e)
    {
      System.out.println(e);
    }
    System.out.println("total_bins:"+total_bins);
    System.out.println("max_bin_size:"+max_bin_size);
    System.out.println("total_cnt:"+total_cnt);
    bins = new int[total_bins];
    surface.setResizable(true);
    size(500,500);
    background(51);
}

Queue<Integer> draw_highlight = new LinkedList<Integer>();
boolean done = false;
void draw()
{
    if(animate==total_cnt)
    {
      if(!done)
      {
        done = true;
        System.out.println("Done!");
      }
      return;
    }
    //System.out.println("Animate:"+animate);
    int current_bin = getdata();
    bins[current_bin]++;
    point(current_bin, bins[current_bin]>=height?(height-1):bins[current_bin]);
    animate++;
    if (draw_highlight.size()>20)
    {
      Integer r = draw_highlight.poll();
      if (r!=null)
      {
        stroke(150,150,150);
        line(r,0,r,bins[r]>=height?(height-1):bins[r]);
      }
    }
    draw_highlight.add(current_bin);
    stroke(100,255,255);
    for (Integer e : draw_highlight)
    {
      point(e,bins[e]>=height?(height-1):bins[e]);
    }
    stroke(0,0,0);
}
