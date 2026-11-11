package tensor

import "fmt"

func Fused1410SiluAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("silu_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.SiLU()
	r = r.Sign()
	return r, nil
}
func Fused1410SiluAddSignName() string { return "silu_add_sign" }
