clear VOCopts

% dataset
if validAct == 1
    VOCopts.dataset='vcoco_action_valid';
else
    VOCopts.dataset='vcoco_relation_valid';
end

% get current directory with forward slashes
%cwd = '/home/mengyong/Detection/VOCdevkit';
cwd = '../..';
% change this path to point to your copy of the PASCAL VOC data
VOCopts.datadir=[cwd '/'];

% change this path to a writable directory for your results
VOCopts.resdir=[VOCopts.datadir  VOCopts.dataset  '/results' '/'];
% change this path to a writable local directory for the example code
% VOCopts.localdir=[cwd '/local/' VOCopts.dataset '/'];

% initialize the test set
 VOCopts.testset='test'; % use test set for final challenge

% initialize main challenge paths
% VOCopts.annopath=[VOCopts.datadir VOCopts.dataset '/Annotations/%s.xml'];
VOCopts.imgsetpath=[VOCopts.datadir VOCopts.dataset '/%s.txt'];
%VOCopts.detrespath=[VOCopts.resdir '%s_det_' VOCopts.testset '_%s.txt'];
VOCopts.detrespath=[VOCopts.resdir '%s.txt'];
VOCopts.testlabelpath=[VOCopts.datadir VOCopts.dataset '/labels/%s/test/%s.txt'];

% initialize the VOC challenge options
VOCopts.classes={...
    'kick', 'read', 'skateboard', 'ski', 'snowboard', 'surf', 'talk_on_phone', 'work_on_computer'};
%VOCopts.classes={...
%    'kick'};

VOCopts.nclasses=length(VOCopts.classes);	  
VOCopts.minoverlap=0.5;

