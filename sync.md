# To add upstream

git clone https://github.com/userName/Repo New_Repo
cd New_Repo
git remote set-url origin https://github.com/userName/New_Repo
git remote add upstream https://github.com/userName/Repo
git push origin master

# To sync with template :

git fetch upstream
git merge upstream/master
git push origin master