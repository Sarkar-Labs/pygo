package tensor

import "fmt"

func Fused1019CeilMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Ceil()
	r = r.Exp()
	return r, nil
}
func Fused1019CeilMatmulExpName() string { return "ceil_matmul_exp" }
