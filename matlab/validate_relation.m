% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);
validAct = 0;
validRelation = 1;   
% initialize VOC options
VOCinit;
mAP = 0.0; 
% train and test detector for each class
for i=1:VOCopts.nclasses
    cls=VOCopts.classes{i};
    [recall,prec,ap]=VOCevalrelation(VOCopts, cls,true);  % compute and display PR
    mAP = mAP + ap; 
    %if i<VOCopts.nclasses
    %    fprintf('press any key to continue with next class...\n');
    %    drawnow;
    %    pause;
    %end
end
mAP = mAP / VOCopts.nclasses;
fprintf('\nmAP = %f \n', mAP);