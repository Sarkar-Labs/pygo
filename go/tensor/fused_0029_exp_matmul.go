package tensor

import "fmt"

func Fused0029ExpMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("exp_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Exp()
	return r, nil
}
func Fused0029ExpMatmulName() string { return "exp_matmul" }
