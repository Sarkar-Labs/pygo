package tensor

import "fmt"

func Fused0584ExpMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	r = r.Cos()
	return r, nil
}
func Fused0584ExpMatmulCosName() string { return "exp_matmul_cos" }
