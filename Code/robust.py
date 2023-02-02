import re
import gzip
import shutil
def parse_single_file(documents_list, file_dir, output_file):
    output_file = open(output_file, 'a')
    with gzip.open(file_dir, 'rb') as input_dir:
        print file_dir
        text_flag = False
        title_flag = False
        body_flag = False
        text = ''
        for line in input_dir:
            line = line.strip()
            if '<DOCNO>' in line:
                docno = line.rstrip('</DOCNO>').lstrip('<DOCNO>').strip()
                #print docno
                assert docno != ''
            elif '</DOC>' in line:
                #print ('came here')
                #print docno
                if docno in documents_list:
                    output_file.write(docno +'<seperator>' + text+'\n')
                text = ''

            elif '<TEXT>' in line:
                text_flag = True

            elif '</TEXT>' in line:
                text_flag = False
            elif ('<TITLE>' in line) and ('</TITLE>' in line):
                title = line.rstrip('</TITLE>').lstrip('<TITLE>').strip()
                TAG_RE = re.compile(r'<[^>]+>')
                line = TAG_RE.sub('', title)
                text += ' ' + line.rstrip('\n')
            elif ('<TITLE>' in line) or ('<title>' in line):
                title_flag = True
            elif ('</TITLE>' in line) or ('</title>' in line):
                title_flag = False
            elif ('<BODY' in line and ('</BODY>' in line)) or ('<body' in line and ('</body>' in line) ):
                body = line.rstrip('</BODY>').lstrip('<BODY>').strip()
                TAG_RE = re.compile(r'<[^>]+>')
                line = TAG_RE.sub('', body)
                text += ' ' + line.rstrip('\n')
            elif ('<BODY' in line) or ('<body' in line):
                body_flag = True
            elif ('</BODY>' in line) or ('</body>' in line):
                body_flag = False
            elif text_flag or title_flag or body_flag:
                TAG_RE = re.compile(r'<[^>]+>')
                line = TAG_RE.sub('', line)
                text += ' ' + line.rstrip('\n')
    output_file.flush()
    output_file.close()

doc_list = []
infiles = []
#../../Web_data/MQ2007/fold1/documents_list.txt
all_infiles = []
with open('documents_list.txt', 'r') as infile:
    for line in infile:
        if line.strip() in doc_list:
            print 'omg'
        doc_list += [line.strip()]
        infile = line.strip().split('-')[0]+'/'+line.strip().split('-')[1]
        all_infiles += [infile]
        if infile not in infiles:
            #print infile
            infiles += [infile]
print len(doc_list)
print len(infiles)
from collections import Counter
infiles_sorted = sorted(Counter(all_infiles).items(), key=lambda l :l[1],reverse=True)
print infiles_sorted[:30]
print len(infiles_sorted)
'''
for infile in infiles:
    with gzip.open(indirectory + infile + '.gz', 'rb') as f_in:
        with open(indirectory + infile + '.txt', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
'''
#indirectory = '../../Web_data/gov2-corpus/'
#outfile = '../../Web_data/document_corpus_sorted.txt'

indirectory = 'gov2-corpus/'
outfile = 'document_corpus_sorted.txt'
output_file = open(outfile, 'w')
output_file.flush()
output_file.close()
#print infiles[:10]
#infiles = infiles[250:2000]
for infile in infiles_sorted:
    parse_single_file(doc_list, indirectory+infile[0]+'.gz', outfile)





