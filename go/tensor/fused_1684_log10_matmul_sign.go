package tensor

import "fmt"

func Fused1684Log10MatmulSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log10_matmul_sign: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log10()
	r = r.Sign()
	return r, nil
}
func Fused1684Log10MatmulSignName() string { return "log10_matmul_sign" }
