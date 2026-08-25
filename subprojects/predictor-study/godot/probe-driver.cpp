#include "core/math/aabb.h"
#include "core/math/vector3.h"
#include <cstdio>

int main() {
    AABB box(Vector3(0, 0, 0), Vector3(1, 1, 1));
    double acc = 0.0;
    for (int frame = 0; frame < 4; ++frame) {
        for (int i = 0; i < 256; ++i) {
            Vector3 v(i * 0.5f + 1.0f, i * 0.25f, i * 0.125f);
            Vector3 n = v.normalized();
            box.expand_to(n);
            acc += n.x + n.y + n.z;
        }
    }
    printf("acc=%f box=%f\n", acc, (double)(box.size.x + box.size.y + box.size.z));
    return 0;
}
