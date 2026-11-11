package tensor

import "fmt"

func Fused1199SignMatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Exp()
	return r, nil
}
func Fused1199SignMatmulExpName() string { return "sign_matmul_exp" }
