package tensor

import "fmt"

func Fused0839CosMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Cos()
	r = r.Exp()
	return r, nil
}
func Fused0839CosMatmulExpName() string { return "cos_matmul_exp" }
