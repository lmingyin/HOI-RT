function [rec,prec,ap] = VOCevaldet(VOCopts, cls,draw)

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
        gt(i).BB=rec(:, 2:5)';
        gt(i).det=false(size(rec, 1),1);            
    end     

    npos=npos+size(rec, 1); 
end
%fprintf(sprintf(VOCopts.detrespath,id,cls));
% load results
%[ids,confidence,b1,b2,b3,b4]=textread(sprintf(VOCopts.detrespath,id,cls),'%s %f %f %f %f %f');
[ids,confidence,b1,b2,b3,b4]=textread(sprintf(VOCopts.detrespath, cls),'%s %f %f %f %f %f');
BB=[b1 b2 b3 b4]';

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
        fprintf('%s: pr: compute: %d/%d\n',cls,d,nd);
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
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
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
%save nmat1 rec1 prec1
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
