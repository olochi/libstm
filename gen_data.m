% gnd1 = gnd(9:14, :);
% fea1 = fea(9:14, :);
% fea2 = sparse(fea1);
% libsvmwrite('Yale_test_w.txt', gnd1, fea2);
% for i =1: 
gnd1 = gnd(1:11, :);
fea1 = fea(1:11, :);
fea2 = sparse(fea1);
libsvmwrite('Yale_train_1.txt', gnd1, fea2);

gnd1 = gnd(7:9, :);
fea1 = fea(7:9, :);
fea2 = sparse(fea1);
libsvmwrite('Yale_test_right.txt', gnd1, fea2);

gnd1 = gnd(20:25, :);
fea1 = fea(20:25, :);
fea2 = sparse(fea1);
libsvmwrite('Yale_test_wrong.txt', gnd1, fea2);