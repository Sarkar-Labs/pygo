package tensor

import "fmt"

func Fused0479SqrtMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sqrt()
	r = r.Exp()
	return r, nil
}
func Fused0479SqrtMatmulExpName() string { return "sqrt_matmul_exp" }
