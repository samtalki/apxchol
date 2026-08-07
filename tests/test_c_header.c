#include "apxchol/c_api.h"

_Static_assert(sizeof(apxchol_status) == 4, "status must have a 32-bit ABI");
_Static_assert(sizeof(((apxchol_solve_info*)0)->iterations) == 4,
               "iteration count must have a 32-bit ABI");

int main(void) {
    apxchol_solve_info info = {0, 0.0, 0.0, 0.0};
    apxchol_status status = APXCHOL_STATUS_SUCCESS;
    return info.iterations + (int)status;
}
