package tensor

import "fmt"

func Fused1189SignMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Abs()
	return r, nil
}
func Fused1189SignMatmulAbsName() string { return "sign_matmul_abs" }
