function [new_pm,new_pfa] = Filter_DET (pm,pfa)
out = 1;
new_pm = pm(1);
new_pfa = pfa(1);

for i=2:length(pm)
  if (pm(i) == new_pm(out))
    continue;
  end
  if (pfa(i) == new_pfa(out))
    continue;
  end

  out = out+1;
  new_pm(out) = pm(i-1);
  new_pfa(out) = pfa(i-1);
end;

out = out+1;
new_pm(out) = pm(end);
new_pfa(out) = pfa(end);
