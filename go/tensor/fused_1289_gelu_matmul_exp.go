package tensor

import "fmt"

func Fused1289GeluMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.GELU()
	r = r.Exp()
	return r, nil
}
func Fused1289GeluMatmulExpName() string { return "gelu_matmul_exp" }
