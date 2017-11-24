function [rec,prec,ap] = VOCevalrelation(VOCopts, cls,draw)

%fprintf(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
% load test set
[gtids, t]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
% load ground truth objects
tic;
npos=0;
gt(length(gtids))=struct('BB', [], 'det', []);
for i=1:length(gtids)
    %fprintf(sprintf(VOCopts.testlabelpath,gtids{i}));
    rec = importdata(sprintf(VOCopts.testlabelpath,cls,gtids{i})); 
    if size(rec, 1)
        gt(i).BB=rec(:, 2:9)';
        gt(i).det=false(size(rec, 1),1);            
    end     

    npos=npos+size(rec, 1); 
end
% load results
[ids,confidence, b1, b2, b3, b4, b5, b6, b7, b8]=textread(sprintf(VOCopts.detrespath,cls),...
                                                '%s %f %f %f %f %f %f %f %f %f');
BB=[b1 b2 b3 b4 b5 b6 b7 b8]';

% sort detections by decreasing confidence
[~,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;
for d=1:nd
    % display progress
    if toc>1
        fprintf('%s: pr: compute: %d/%d\n', cls, d, nd);
        drawnow;
        tic;
    end
    
    % find ground truth image
    i=strmatch(ids{d},gtids,'exact');
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end

    % assign detection to ground truth object if any
    bb=BB(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi_act=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        bi_obj=[max(bb(5),bbgt(5)) ; max(bb(6),bbgt(6)) ; min(bb(7),bbgt(7)) ; min(bb(8),bbgt(8))];
        iw_act=bi_act(3)-bi_act(1)+1;
        iw_obj=bi_obj(3)-bi_obj(1)+1;
        ih_act=bi_act(4)-bi_act(2)+1;
        ih_obj=bi_obj(4)-bi_obj(2)+1;
        if iw_act>0 & ih_act>0 & iw_obj>0 & ih_obj>0               
            % compute overlap as area of intersection / area of union
            ua_act=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw_act*ih_act;
            ov_act=iw_act*ih_act/ua_act;

            ua_obj=(bb(7)-bb(5)+1)*(bb(8)-bb(6)+1)+...
               (bbgt(7)-bbgt(5)+1)*(bbgt(8)-bbgt(6)+1)-...
               iw_obj*ih_obj;
            ov_obj=iw_obj*ih_obj/ua_obj;            
            ov = min([ov_act,ov_obj]);
            if ov > ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=VOCopts.minoverlap
        if ~gt(i).det(jmax) %assert the true box not matched before
            tp(d)=1;            % true positive
            gt(i).det(jmax)=true; %one true box match once only
        else
            fp(d)=1;            % false positive (multiple detection)
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall
fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/npos;
prec=tp./(fp+tp);
ap = VOCap(rec,prec);
%ap1 = VOCap(rec1,prec1);
if draw
    % plot precision/recall
    plot(rec,prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, subset: %s, AP = %.3f',cls,VOCopts.testset,ap));
    saveas(gcf, cls, 'jpg')
end
