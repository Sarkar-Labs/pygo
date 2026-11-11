package tensor

import "fmt"

func Fused0324TanhMatmulCeil(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_ceil: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Ceil()
	return r, nil
}
func Fused0324TanhMatmulCeilName() string { return "tanh_matmul_ceil" }
