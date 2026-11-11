package tensor

import "fmt"

func Fused1044CeilMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Ceil()
	return r, nil
}
func Fused1044CeilMatmulCeilName() string { return "ceil_matmul_ceil" }
