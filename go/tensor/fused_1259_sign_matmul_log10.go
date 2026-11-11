package tensor

import "fmt"

func Fused1259SignMatmulLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_matmul_log10: nil operand")
	}
	r := x.MatMul(y)
	r = r.Sign()
	r = r.Log10()
	return r, nil
}
func Fused1259SignMatmulLog10Name() string { return "sign_matmul_log10" }
