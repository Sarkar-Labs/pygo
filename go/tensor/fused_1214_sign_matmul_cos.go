package tensor

import "fmt"

func Fused1214SignMatmulCos(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_cos: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Cos()
	return r, nil
}
func Fused1214SignMatmulCosName() string { return "sign_matmul_cos" }
