git config --global core.pager ""

git reset --hard 3dcbf4c2ec78c03cb8e96b69e4fb9f65f3b7fa24

git checkout -b restore-point 3dcbf4c2ec78c03cb8e96b69e4fb9f65f3b7fa24

git revert 3dcbf4c2ec78c03cb8e96b69e4fb9f65f3b7fa24..HEAD

