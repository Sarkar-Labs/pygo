package tensor

import "fmt"

func Fused1256SignSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sign_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Sign()
	r = r.Log10()
	return r, nil
}
func Fused1256SignSubLog10Name() string { return "sign_sub_log10" }
