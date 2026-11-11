package tensor

import "fmt"

func Fused0749SinMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sin_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sin()
	r = r.Exp()
	return r, nil
}
func Fused0749SinMatmulExpName() string { return "sin_matmul_exp" }
