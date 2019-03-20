/*
 * bndldstx log processor
 * 2017 Tong Zhang<ztong@vt.edu>
 */
#include <iostream>
#include <fstream>

#include <map>


#define PTR_TO_MPX_PAGE(x) \
    ((x>>7) & 0x01FFFFFFFFFFFFFFUL)

using namespace std;

void update_progress(int want_thresh)
{
    static int thresh = 0;
    static char prog = 0;
    static char px[] = {'|','/','-','\\','|','/','-','\\'};
    thresh = (thresh+1)%want_thresh;
    if (thresh)
        return;
    prog = (prog+1)%8;
    printf("\x8%c",px[prog]);
    fflush(stdout);
}

void usage(char* app)
{
    cerr<<app<<" infile praw finalbin\n"
        <<"infile - input file\n"
        <<"praw - processed raw file, for Processing 3 script input\n"
        <<"finalbin - the finalbin'd result\n";
    exit(-1);
}

int main(int argc, char** argv)
{
    if (argc!=4)
    {
        usage(argv[0]);
    }
    ifstream infile;
    infile.open(argv[1], ios::in);
    ofstream praw;
    praw.open(argv[2], ios::trunc);
    ofstream finalbin;
    finalbin.open(argv[3], ios::trunc);

    map<unsigned long, size_t> mpx_page_ref_cnt;
    map<unsigned long, size_t> mpx_page_hash;//page->bin number
    //read and parse ptr
    unsigned long ptr;
    size_t total_bin = 0;
    string line;
    cout<<"Collecting stat... ";
    for(line;getline(infile,line);)
    {
        size_t pos = line.find("bnd")+7;
        string x = line.substr(pos);
        sscanf(x.c_str(),"0x%lx", &ptr);
        unsigned long page = PTR_TO_MPX_PAGE(ptr);
        auto cnt = mpx_page_ref_cnt.find(page);
        if(cnt==mpx_page_ref_cnt.end())
        {
            mpx_page_ref_cnt[page] = 1;
            mpx_page_hash[page] = total_bin;
            total_bin++;
        }else
        {
            mpx_page_ref_cnt[page] = cnt->second+1;
        }
        praw<<mpx_page_hash[page]<<"\n";
        update_progress(10000);
    }
    //append total bins and max bin size
    praw<<total_bin<<"\n";

    cout<<"Done!\n"
        <<"Writing output... ";
    size_t max_bin_size = 0;
    for(auto pn = mpx_page_ref_cnt.begin(),
            pe = mpx_page_ref_cnt.end();
            pn!=pe; ++pn)
    {
        if (max_bin_size<pn->second)
            max_bin_size = pn->second;
        finalbin<<"0x"<<hex<<pn->first<<dec
                <<","<<pn->second<<endl;
        update_progress(10000);
    }
    cout<<"Done!\n";

    praw<<max_bin_size<<"\n";

    infile.close();
    praw.close();
    finalbin.close();

    return 0;
}

