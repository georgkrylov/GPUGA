Brief description:
This is a working version of the software that was used in the research for 
synthesis of quantum circuits.To read more in details refer to 
https://ieeexplore.ieee.org/document/7964993/

Compilation instructions:

$cd Release
$make

Execution instructions:
Make sure parameters of algorithm (contained in the parameters.h) are matching 
the target defined in Target.txt. If they do not, after modification of 
parameters.h you will have to run

$cd Release
$make clean
$make

After parameters are verified and new executable was compiled, you can run

chmod +x LaunchTheSearch.sh
. LaunchTheSearch.sh

Verbose parameter:
If you want to use the application in background, just set the verbose parameter
to 0. This will prevent all output to console except initializaiton.

Description of Target.txt
Target.txt consists of target matrix terms in a format of 
{real imaginary}, without braces. The total number of matrix terms is
2^numberOfWires